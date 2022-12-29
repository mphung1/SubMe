import { ReactNode } from "react";
import NextLink from "next/link";
import {
  Container,
  Box,
  Link,
  Stack,
  Text,
  Heading,
  Flex,
  Spacer,
  Menu,
  MenuItem,
  MenuList,
  MenuButton,
  IconButton,
} from "@chakra-ui/react";
import { HamburgerIcon } from "@chakra-ui/icons";

const LinkItem = ({
  href,
  path,
  children,
}: {
  href: string;
  path: string;
  children: ReactNode;
}) => {
  const active = path == href;
  return (
    <NextLink href={href}>
      <Link
        p={2}
        bg={active ? "glassTeal" : undefined}
        color="whiteAlpha.900"
      >
        {children}
      </Link>
    </NextLink>
  );
};

const Navbar = (props) => {
  const { path } = props;

  return (
    <Box
      position="fixed"
      as="nav"
      w="100%"
      bg="black"
      zIndex={1}
      color="black"
      {...props}
    >
      <Flex minWidth='max-content' alignItems='center'>
        <LinkItem href="/" path={path}>
          <Heading size="md" ml="5">
            SubMe
          </Heading>
        </LinkItem>
        <Stack
          direction={{ base: "column", md: "row" }}
          display={{ base: "none", md: "flex" }}
          pos="absolute"
          left="80%"
        >
          <LinkItem href="/" path={path}>
            Docs
          </LinkItem>
          <LinkItem href="/demo" path={path}>
            Application Demo
          </LinkItem>
        </Stack>

        <Box flex={1}>
          <Box display={{ base: "inline-block", md: "none" }}>
            <Menu isLazy id="navbar-menu">
              <MenuButton
                as={IconButton}
                icon={<HamburgerIcon />}
                variant="outline"
                aria-label="Options"
                color="white"
                pos="absolute"
                left="90%"
                top="0"
              />
              <MenuList>
                <NextLink href="/" passHref>
                  <MenuItem as={Link}>Docs</MenuItem>
                </NextLink>
                <NextLink href="/demo" passHref>
                  <MenuItem as={Link}> Application Demo </MenuItem>
                </NextLink>
              </MenuList>
            </Menu>
          </Box>
        </Box>
      </Flex>
    </Box>
  );
};

export default Navbar;
